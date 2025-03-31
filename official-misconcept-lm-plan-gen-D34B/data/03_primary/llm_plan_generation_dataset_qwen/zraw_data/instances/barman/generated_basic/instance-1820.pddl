(define (problem prob)
 (:domain barman)
 (:objects 
      shaker171 - shaker
      left right - hand
      shot263 shot52 shot136 - shot
      ingredient396 ingredient128 ingredient327 ingredient255 - ingredient
      cocktail82 - cocktail
      dispenser353 dispenser88 dispenser439 dispenser221 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker171)
  (ontable shot263)
  (ontable shot52)
  (ontable shot136)
  (dispenses dispenser353 ingredient396)
  (dispenses dispenser88 ingredient128)
  (dispenses dispenser439 ingredient327)
  (dispenses dispenser221 ingredient255)
  (clean shaker171)
  (clean shot263)
  (clean shot52)
  (clean shot136)
  (empty shaker171)
  (empty shot263)
  (empty shot52)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker171 l0)
  (shaker-level shaker171 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail82 ingredient255)
  (cocktail-part2 cocktail82 ingredient327)
)
 (:goal
  (and
      (contains shot263 cocktail82)
      (contains shot52 cocktail82)
)))
