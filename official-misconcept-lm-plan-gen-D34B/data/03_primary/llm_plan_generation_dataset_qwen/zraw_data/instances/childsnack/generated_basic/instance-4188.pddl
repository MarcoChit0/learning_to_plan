; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 939829

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child243 child89 - child
    bread197 bread334 - bread-portion
    content321 content318 - content-portion
    tray110 - tray
    table397 table204 table346 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray110 kitchen)
     (at_kitchen_bread bread197)
     (at_kitchen_bread bread334)
     (at_kitchen_content content321)
     (at_kitchen_content content318)
     (not_allergic_gluten child89)
     (not_allergic_gluten child243)
     (waiting child243 table346)
     (waiting child89 table204)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child243)
     (served child89)
    )
  )
)
