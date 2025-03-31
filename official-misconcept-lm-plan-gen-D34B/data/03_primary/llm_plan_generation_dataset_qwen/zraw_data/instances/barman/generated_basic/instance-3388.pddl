(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot363 shot122 shot439 - shot
      ingredient0 ingredient251 ingredient189 - ingredient
      cocktail368 - cocktail
      dispenser247 dispenser377 dispenser482 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot363)
  (ontable shot122)
  (ontable shot439)
  (dispenses dispenser247 ingredient0)
  (dispenses dispenser377 ingredient251)
  (dispenses dispenser482 ingredient189)
  (clean shaker189)
  (clean shot363)
  (clean shot122)
  (clean shot439)
  (empty shaker189)
  (empty shot363)
  (empty shot122)
  (empty shot439)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail368 ingredient189)
  (cocktail-part2 cocktail368 ingredient0)
)
 (:goal
  (and
      (contains shot363 cocktail368)
      (contains shot122 cocktail368)
)))
