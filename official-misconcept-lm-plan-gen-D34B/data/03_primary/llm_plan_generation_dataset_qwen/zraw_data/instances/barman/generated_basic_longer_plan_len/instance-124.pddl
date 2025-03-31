(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot366 shot243 shot159 - shot
      ingredient26 ingredient112 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser444 dispenser96 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot366)
  (ontable shot243)
  (ontable shot159)
  (dispenses dispenser444 ingredient26)
  (dispenses dispenser96 ingredient112)
  (clean shaker143)
  (clean shot366)
  (clean shot243)
  (clean shot159)
  (empty shaker143)
  (empty shot366)
  (empty shot243)
  (empty shot159)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient26)
  (cocktail-part2 cocktail1 ingredient112)
  (cocktail-part1 cocktail2 ingredient112)
  (cocktail-part2 cocktail2 ingredient26)
)
 (:goal
  (and
      (contains shot366 cocktail1)
      (contains shot243 cocktail2)
)))
