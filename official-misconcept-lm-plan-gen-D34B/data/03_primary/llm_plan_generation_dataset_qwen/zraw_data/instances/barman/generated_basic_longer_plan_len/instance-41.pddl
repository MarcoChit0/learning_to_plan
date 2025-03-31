(define (problem prob)
 (:domain barman)
 (:objects 
      shaker88 - shaker
      left right - hand
      shot172 shot142 shot337 - shot
      ingredient70 ingredient244 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser157 dispenser455 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker88)
  (ontable shot172)
  (ontable shot142)
  (ontable shot337)
  (dispenses dispenser157 ingredient70)
  (dispenses dispenser455 ingredient244)
  (clean shaker88)
  (clean shot172)
  (clean shot142)
  (clean shot337)
  (empty shaker88)
  (empty shot172)
  (empty shot142)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker88 l0)
  (shaker-level shaker88 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient70)
  (cocktail-part2 cocktail1 ingredient244)
  (cocktail-part1 cocktail2 ingredient244)
  (cocktail-part2 cocktail2 ingredient70)
)
 (:goal
  (and
      (contains shot172 cocktail1)
      (contains shot142 cocktail2)
)))
