; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 882778

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child488 child48 - child
    bread65 bread173 - bread-portion
    content285 content297 - content-portion
    tray110 tray330 tray17 tray296 - tray
    table219 table465 table204 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray110 kitchen)
     (at tray330 kitchen)
     (at tray17 kitchen)
     (at tray296 kitchen)
     (at_kitchen_bread bread65)
     (at_kitchen_bread bread173)
     (at_kitchen_content content285)
     (at_kitchen_content content297)
     (not_allergic_gluten child488)
     (not_allergic_gluten child48)
     (waiting child488 table204)
     (waiting child48 table204)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child488)
     (served child48)
    )
  )
)
