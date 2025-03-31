; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 351280

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child28 child442 - child
    bread231 bread28 - bread-portion
    content233 content37 - content-portion
    tray388 tray397 tray17 - tray
    table160 table367 table446 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray388 kitchen)
     (at tray397 kitchen)
     (at tray17 kitchen)
     (at_kitchen_bread bread231)
     (at_kitchen_bread bread28)
     (at_kitchen_content content233)
     (at_kitchen_content content37)
     (no_gluten_bread bread231)
     (no_gluten_content content233)
     (allergic_gluten child442)
     (not_allergic_gluten child28)
     (waiting child28 table446)
     (waiting child442 table446)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child28)
     (served child442)
    )
  )
)
