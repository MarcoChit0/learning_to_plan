(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot76 shot247 shot488 - shot
      ingredient137 ingredient149 ingredient100 ingredient417 - ingredient
      cocktail1 - cocktail
      dispenser298 dispenser119 dispenser44 dispenser262 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot76)
  (ontable shot247)
  (ontable shot488)
  (dispenses dispenser298 ingredient137)
  (dispenses dispenser119 ingredient149)
  (dispenses dispenser44 ingredient100)
  (dispenses dispenser262 ingredient417)
  (clean shaker185)
  (clean shot76)
  (clean shot247)
  (clean shot488)
  (empty shaker185)
  (empty shot76)
  (empty shot247)
  (empty shot488)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient137)
  (cocktail-part2 cocktail1 ingredient100)
)
 (:goal
  (and
      (contains shot76 cocktail1)
      (contains shot247 cocktail1)
)))
