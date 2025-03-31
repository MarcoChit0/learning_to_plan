(define (problem prob)
 (:domain barman)
 (:objects 
      shaker432 - shaker
      left right - hand
      shot267 shot79 shot151 - shot
      ingredient458 ingredient86 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser177 dispenser87 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker432)
  (ontable shot267)
  (ontable shot79)
  (ontable shot151)
  (dispenses dispenser177 ingredient458)
  (dispenses dispenser87 ingredient86)
  (clean shaker432)
  (clean shot267)
  (clean shot79)
  (clean shot151)
  (empty shaker432)
  (empty shot267)
  (empty shot79)
  (empty shot151)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker432 l0)
  (shaker-level shaker432 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient458)
  (cocktail-part2 cocktail1 ingredient86)
  (cocktail-part1 cocktail2 ingredient86)
  (cocktail-part2 cocktail2 ingredient458)
)
 (:goal
  (and
      (contains shot267 cocktail1)
      (contains shot79 cocktail2)
)))
