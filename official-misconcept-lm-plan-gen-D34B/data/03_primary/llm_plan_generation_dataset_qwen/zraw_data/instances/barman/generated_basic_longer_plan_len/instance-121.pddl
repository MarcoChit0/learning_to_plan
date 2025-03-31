(define (problem prob)
 (:domain barman)
 (:objects 
      shaker159 - shaker
      left right - hand
      shot268 shot147 shot177 - shot
      ingredient193 ingredient270 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser444 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker159)
  (ontable shot268)
  (ontable shot147)
  (ontable shot177)
  (dispenses dispenser444 ingredient193)
  (dispenses dispenser383 ingredient270)
  (clean shaker159)
  (clean shot268)
  (clean shot147)
  (clean shot177)
  (empty shaker159)
  (empty shot268)
  (empty shot147)
  (empty shot177)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker159 l0)
  (shaker-level shaker159 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient270)
  (cocktail-part2 cocktail1 ingredient193)
  (cocktail-part1 cocktail2 ingredient270)
  (cocktail-part2 cocktail2 ingredient193)
)
 (:goal
  (and
      (contains shot268 cocktail2)
      (contains shot147 cocktail1)
)))
