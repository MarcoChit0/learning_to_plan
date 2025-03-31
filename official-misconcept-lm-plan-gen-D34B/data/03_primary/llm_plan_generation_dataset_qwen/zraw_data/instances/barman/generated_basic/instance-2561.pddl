(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot8 shot62 shot487 - shot
      ingredient181 ingredient485 - ingredient
      cocktail338 - cocktail
      dispenser469 dispenser495 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot8)
  (ontable shot62)
  (ontable shot487)
  (dispenses dispenser469 ingredient181)
  (dispenses dispenser495 ingredient485)
  (clean shaker251)
  (clean shot8)
  (clean shot62)
  (clean shot487)
  (empty shaker251)
  (empty shot8)
  (empty shot62)
  (empty shot487)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail338 ingredient181)
  (cocktail-part2 cocktail338 ingredient485)
)
 (:goal
  (and
      (contains shot8 cocktail338)
      (contains shot62 ingredient181)
)))
