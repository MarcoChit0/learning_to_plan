(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot404 shot296 shot80 - shot
      ingredient449 ingredient341 - ingredient
      cocktail5 - cocktail
      dispenser22 dispenser451 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot404)
  (ontable shot296)
  (ontable shot80)
  (dispenses dispenser22 ingredient449)
  (dispenses dispenser451 ingredient341)
  (clean shaker251)
  (clean shot404)
  (clean shot296)
  (clean shot80)
  (empty shaker251)
  (empty shot404)
  (empty shot296)
  (empty shot80)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient341)
  (cocktail-part2 cocktail5 ingredient449)
)
 (:goal
  (and
      (contains shot404 cocktail5)
      (contains shot296 ingredient341)
)))
