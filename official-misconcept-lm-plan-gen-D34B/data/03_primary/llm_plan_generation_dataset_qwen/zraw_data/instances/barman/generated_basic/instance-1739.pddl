(define (problem prob)
 (:domain barman)
 (:objects 
      shaker92 - shaker
      left right - hand
      shot425 shot183 - shot
      ingredient193 ingredient221 ingredient364 - ingredient
      cocktail44 - cocktail
      dispenser175 dispenser284 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker92)
  (ontable shot425)
  (ontable shot183)
  (dispenses dispenser175 ingredient193)
  (dispenses dispenser284 ingredient221)
  (dispenses dispenser417 ingredient364)
  (clean shaker92)
  (clean shot425)
  (clean shot183)
  (empty shaker92)
  (empty shot425)
  (empty shot183)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker92 l0)
  (shaker-level shaker92 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail44 ingredient221)
  (cocktail-part2 cocktail44 ingredient364)
)
 (:goal
  (and
      (contains shot425 cocktail44)
)))
