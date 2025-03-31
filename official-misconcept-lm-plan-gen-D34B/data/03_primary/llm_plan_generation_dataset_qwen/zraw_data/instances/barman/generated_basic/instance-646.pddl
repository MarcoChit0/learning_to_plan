(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot154 shot328 - shot
      ingredient416 ingredient254 ingredient359 - ingredient
      cocktail1 - cocktail
      dispenser313 dispenser173 dispenser215 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot154)
  (ontable shot328)
  (dispenses dispenser313 ingredient416)
  (dispenses dispenser173 ingredient254)
  (dispenses dispenser215 ingredient359)
  (clean shaker151)
  (clean shot154)
  (clean shot328)
  (empty shaker151)
  (empty shot154)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient416)
  (cocktail-part2 cocktail1 ingredient359)
)
 (:goal
  (and
      (contains shot154 cocktail1)
)))
