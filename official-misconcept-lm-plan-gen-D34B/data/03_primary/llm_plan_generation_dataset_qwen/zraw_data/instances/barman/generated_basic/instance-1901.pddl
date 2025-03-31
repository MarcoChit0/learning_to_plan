(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot498 shot367 shot283 shot431 - shot
      ingredient121 ingredient288 - ingredient
      cocktail407 - cocktail
      dispenser188 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot498)
  (ontable shot367)
  (ontable shot283)
  (ontable shot431)
  (dispenses dispenser188 ingredient121)
  (dispenses dispenser49 ingredient288)
  (clean shaker447)
  (clean shot498)
  (clean shot367)
  (clean shot283)
  (clean shot431)
  (empty shaker447)
  (empty shot498)
  (empty shot367)
  (empty shot283)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail407 ingredient121)
  (cocktail-part2 cocktail407 ingredient288)
)
 (:goal
  (and
      (contains shot498 cocktail407)
      (contains shot367 cocktail407)
      (contains shot283 ingredient121)
)))
