; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 153389

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child127 child190 - child
    bread16 bread347 - bread-portion
    content80 content249 - content-portion
    tray258 - tray
    table462 table24 table127 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray258 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_bread bread347)
     (at_kitchen_content content80)
     (at_kitchen_content content249)
     (no_gluten_bread bread16)
     (no_gluten_content content80)
     (allergic_gluten child127)
     (not_allergic_gluten child190)
     (waiting child127 table127)
     (waiting child190 table462)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child127)
     (served child190)
    )
  )
)
