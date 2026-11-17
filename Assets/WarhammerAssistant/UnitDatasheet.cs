using System;

namespace WarhammerAssistant
{
    [Serializable]
    internal sealed class UnitCatalog
    {
        public UnitEntry[] units;
    }

    [Serializable]
    internal sealed class UnitEntry
    {
        public string label;
        public string name;
        public string display_name;
        public string faction;
        public string description;
        public bool legends;

        public string Title => string.IsNullOrWhiteSpace(display_name) ? name : display_name;
    }

    [Serializable]
    internal sealed class DatasheetCatalog
    {
        public UnitDatasheet[] datasheets;
    }

    [Serializable]
    internal sealed class UnitDatasheet
    {
        public string label;
        public string version;
        public UnitStats stats;
        public WeaponProfile[] weapons;
        public AbilityProfile[] abilities;
    }

    [Serializable]
    internal sealed class UnitStats
    {
        public string movement;
        public string toughness;
        public string save;
        public string wounds;
        public string leadership;
        public string objectiveControl;
        public string invulnerableSave;
    }

    [Serializable]
    internal sealed class WeaponProfile
    {
        public string name;
        public string range;
        public string attacks;
        public string skill;
        public string strength;
        public string armourPenetration;
        public string damage;
        public string keywords;
    }

    [Serializable]
    internal sealed class AbilityProfile
    {
        public string name;
        public string description;
    }
}
