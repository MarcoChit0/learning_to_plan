(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot99 shot36 shot401 - shot
      ingredient149 ingredient51 ingredient32 ingredient396 - ingredient
      cocktail266 - cocktail
      dispenser412 dispenser155 dispenser477 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot99)
  (ontable shot36)
  (ontable shot401)
  (dispenses dispenser412 ingredient149)
  (dispenses dispenser155 ingredient51)
  (dispenses dispenser477 ingredient32)
  (dispenses dispenser166 ingredient396)
  (clean shaker350)
  (clean shot99)
  (clean shot36)
  (clean shot401)
  (empty shaker350)
  (empty shot99)
  (empty shot36)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail266 ingredient396)
  (cocktail-part2 cocktail266 ingredient32)
)
 (:goal
  (and
      (contains shot99 cocktail266)
      (contains shot36 ingredient51)
)))
