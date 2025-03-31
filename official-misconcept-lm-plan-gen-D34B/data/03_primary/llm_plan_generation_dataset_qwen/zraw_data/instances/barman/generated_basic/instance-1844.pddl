(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot183 shot241 - shot
      ingredient51 ingredient151 ingredient288 - ingredient
      cocktail373 - cocktail
      dispenser385 dispenser468 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot183)
  (ontable shot241)
  (dispenses dispenser385 ingredient51)
  (dispenses dispenser468 ingredient151)
  (dispenses dispenser370 ingredient288)
  (clean shaker42)
  (clean shot183)
  (clean shot241)
  (empty shaker42)
  (empty shot183)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail373 ingredient151)
  (cocktail-part2 cocktail373 ingredient288)
)
 (:goal
  (and
      (contains shot183 cocktail373)
)))
