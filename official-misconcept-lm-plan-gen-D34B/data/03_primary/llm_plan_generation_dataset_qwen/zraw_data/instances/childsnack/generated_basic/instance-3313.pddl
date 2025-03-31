; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.3
; random seed: 856864

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child472 child125 - child
    bread483 bread294 - bread-portion
    content156 content446 - content-portion
    tray39 tray19 - tray
    table97 table485 table149 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray39 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread483)
     (at_kitchen_bread bread294)
     (at_kitchen_content content156)
     (at_kitchen_content content446)
     (not_allergic_gluten child472)
     (not_allergic_gluten child125)
     (waiting child472 table485)
     (waiting child125 table149)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child472)
     (served child125)
    )
  )
)
