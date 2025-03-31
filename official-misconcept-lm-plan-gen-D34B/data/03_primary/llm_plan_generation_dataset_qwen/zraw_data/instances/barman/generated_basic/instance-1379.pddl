(define (problem prob)
 (:domain barman)
 (:objects 
      shaker155 - shaker
      left right - hand
      shot438 shot374 shot492 - shot
      ingredient23 ingredient310 ingredient333 ingredient280 - ingredient
      cocktail332 - cocktail
      dispenser71 dispenser260 dispenser187 dispenser10 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker155)
  (ontable shot438)
  (ontable shot374)
  (ontable shot492)
  (dispenses dispenser71 ingredient23)
  (dispenses dispenser260 ingredient310)
  (dispenses dispenser187 ingredient333)
  (dispenses dispenser10 ingredient280)
  (clean shaker155)
  (clean shot438)
  (clean shot374)
  (clean shot492)
  (empty shaker155)
  (empty shot438)
  (empty shot374)
  (empty shot492)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker155 l0)
  (shaker-level shaker155 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail332 ingredient333)
  (cocktail-part2 cocktail332 ingredient310)
)
 (:goal
  (and
      (contains shot438 cocktail332)
      (contains shot374 cocktail332)
)))
