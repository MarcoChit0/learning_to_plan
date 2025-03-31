; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 663405

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child359 child206 - child
    bread426 bread338 - bread-portion
    content242 content444 - content-portion
    tray70 tray162 tray161 - tray
    table105 table266 table36 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray162 kitchen)
     (at tray161 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread338)
     (at_kitchen_content content242)
     (at_kitchen_content content444)
     (not_allergic_gluten child206)
     (not_allergic_gluten child359)
     (waiting child359 table105)
     (waiting child206 table266)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child359)
     (served child206)
    )
  )
)
