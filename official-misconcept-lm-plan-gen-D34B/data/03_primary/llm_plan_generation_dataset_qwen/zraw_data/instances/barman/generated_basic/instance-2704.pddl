(define (problem prob)
 (:domain barman)
 (:objects 
      shaker476 - shaker
      left right - hand
      shot187 shot169 shot136 - shot
      ingredient319 ingredient371 - ingredient
      cocktail477 - cocktail
      dispenser174 dispenser434 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker476)
  (ontable shot187)
  (ontable shot169)
  (ontable shot136)
  (dispenses dispenser174 ingredient319)
  (dispenses dispenser434 ingredient371)
  (clean shaker476)
  (clean shot187)
  (clean shot169)
  (clean shot136)
  (empty shaker476)
  (empty shot187)
  (empty shot169)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker476 l0)
  (shaker-level shaker476 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail477 ingredient371)
  (cocktail-part2 cocktail477 ingredient319)
)
 (:goal
  (and
      (contains shot187 cocktail477)
      (contains shot169 ingredient319)
)))
