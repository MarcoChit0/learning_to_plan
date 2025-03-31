(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot384 - shot
      ingredient273 ingredient384 ingredient396 - ingredient
      cocktail228 - cocktail
      dispenser331 dispenser187 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot384)
  (dispenses dispenser331 ingredient273)
  (dispenses dispenser187 ingredient384)
  (dispenses dispenser49 ingredient396)
  (clean shaker4)
  (clean shot384)
  (empty shaker4)
  (empty shot384)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail228 ingredient384)
  (cocktail-part2 cocktail228 ingredient273)
)
 (:goal
  (and
      (contains shot384 cocktail228)
)))
