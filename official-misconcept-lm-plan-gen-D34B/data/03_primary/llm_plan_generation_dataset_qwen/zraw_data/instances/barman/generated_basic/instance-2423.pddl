(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot201 shot143 shot200 shot294 - shot
      ingredient264 ingredient50 - ingredient
      cocktail82 - cocktail
      dispenser173 dispenser40 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot201)
  (ontable shot143)
  (ontable shot200)
  (ontable shot294)
  (dispenses dispenser173 ingredient264)
  (dispenses dispenser40 ingredient50)
  (clean shaker170)
  (clean shot201)
  (clean shot143)
  (clean shot200)
  (clean shot294)
  (empty shaker170)
  (empty shot201)
  (empty shot143)
  (empty shot200)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail82 ingredient264)
  (cocktail-part2 cocktail82 ingredient50)
)
 (:goal
  (and
      (contains shot201 cocktail82)
      (contains shot143 ingredient50)
      (contains shot200 cocktail82)
)))
