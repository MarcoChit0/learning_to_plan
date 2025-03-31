(define (problem prob)
 (:domain barman)
 (:objects 
      shaker254 - shaker
      left right - hand
      shot366 shot94 shot4 - shot
      ingredient253 ingredient186 ingredient11 - ingredient
      cocktail63 - cocktail
      dispenser313 dispenser200 dispenser404 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker254)
  (ontable shot366)
  (ontable shot94)
  (ontable shot4)
  (dispenses dispenser313 ingredient253)
  (dispenses dispenser200 ingredient186)
  (dispenses dispenser404 ingredient11)
  (clean shaker254)
  (clean shot366)
  (clean shot94)
  (clean shot4)
  (empty shaker254)
  (empty shot366)
  (empty shot94)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker254 l0)
  (shaker-level shaker254 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail63 ingredient186)
  (cocktail-part2 cocktail63 ingredient11)
)
 (:goal
  (and
      (contains shot366 cocktail63)
      (contains shot94 ingredient253)
)))
