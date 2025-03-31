; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 683264

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child232 child197 - child
    bread305 bread268 - bread-portion
    content273 content289 - content-portion
    tray232 tray385 - tray
    table378 table42 table477 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray232 kitchen)
     (at tray385 kitchen)
     (at_kitchen_bread bread305)
     (at_kitchen_bread bread268)
     (at_kitchen_content content273)
     (at_kitchen_content content289)
     (not_allergic_gluten child232)
     (not_allergic_gluten child197)
     (waiting child232 table477)
     (waiting child197 table378)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child232)
     (served child197)
    )
  )
)
