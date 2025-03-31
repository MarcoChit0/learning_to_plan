(define (problem prob)
 (:domain barman)
 (:objects 
      shaker65 - shaker
      left right - hand
      shot183 shot325 - shot
      ingredient98 ingredient198 ingredient262 - ingredient
      cocktail209 - cocktail
      dispenser317 dispenser323 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker65)
  (ontable shot183)
  (ontable shot325)
  (dispenses dispenser317 ingredient98)
  (dispenses dispenser323 ingredient198)
  (dispenses dispenser417 ingredient262)
  (clean shaker65)
  (clean shot183)
  (clean shot325)
  (empty shaker65)
  (empty shot183)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker65 l0)
  (shaker-level shaker65 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail209 ingredient262)
  (cocktail-part2 cocktail209 ingredient98)
)
 (:goal
  (and
      (contains shot183 cocktail209)
)))
