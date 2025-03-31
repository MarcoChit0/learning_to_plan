(define (problem prob)
 (:domain barman)
 (:objects 
      shaker137 - shaker
      left right - hand
      shot30 shot249 - shot
      ingredient485 ingredient211 ingredient391 - ingredient
      cocktail100 - cocktail
      dispenser230 dispenser251 dispenser278 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker137)
  (ontable shot30)
  (ontable shot249)
  (dispenses dispenser230 ingredient485)
  (dispenses dispenser251 ingredient211)
  (dispenses dispenser278 ingredient391)
  (clean shaker137)
  (clean shot30)
  (clean shot249)
  (empty shaker137)
  (empty shot30)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker137 l0)
  (shaker-level shaker137 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail100 ingredient391)
  (cocktail-part2 cocktail100 ingredient485)
)
 (:goal
  (and
      (contains shot30 cocktail100)
)))
