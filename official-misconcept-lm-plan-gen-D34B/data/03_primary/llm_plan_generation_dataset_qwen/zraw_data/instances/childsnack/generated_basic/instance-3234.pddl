; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 798969

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child90 child206 - child
    bread89 bread215 - bread-portion
    content288 content37 - content-portion
    tray256 tray385 tray146 tray450 - tray
    table50 table200 table305 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray256 kitchen)
     (at tray385 kitchen)
     (at tray146 kitchen)
     (at tray450 kitchen)
     (at_kitchen_bread bread89)
     (at_kitchen_bread bread215)
     (at_kitchen_content content288)
     (at_kitchen_content content37)
     (no_gluten_bread bread89)
     (no_gluten_content content288)
     (allergic_gluten child90)
     (not_allergic_gluten child206)
     (waiting child90 table50)
     (waiting child206 table200)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child90)
     (served child206)
    )
  )
)
