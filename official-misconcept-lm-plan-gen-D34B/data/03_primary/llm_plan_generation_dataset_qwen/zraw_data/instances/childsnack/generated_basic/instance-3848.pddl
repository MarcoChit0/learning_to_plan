; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 758300

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child307 - child
    bread444 - bread-portion
    content338 - content-portion
    tray17 tray171 tray37 tray238 - tray
    table30 table188 table51 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray17 kitchen)
     (at tray171 kitchen)
     (at tray37 kitchen)
     (at tray238 kitchen)
     (at_kitchen_bread bread444)
     (at_kitchen_content content338)
     (not_allergic_gluten child307)
     (waiting child307 table188)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child307)
    )
  )
)
