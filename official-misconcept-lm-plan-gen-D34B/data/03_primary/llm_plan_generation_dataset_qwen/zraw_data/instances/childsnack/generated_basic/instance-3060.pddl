; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 710090

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child43 child64 - child
    bread410 bread422 - bread-portion
    content227 content296 - content-portion
    tray126 tray320 tray70 - tray
    table405 table317 table244 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at tray320 kitchen)
     (at tray70 kitchen)
     (at_kitchen_bread bread410)
     (at_kitchen_bread bread422)
     (at_kitchen_content content227)
     (at_kitchen_content content296)
     (no_gluten_bread bread422)
     (no_gluten_content content227)
     (allergic_gluten child43)
     (not_allergic_gluten child64)
     (waiting child43 table405)
     (waiting child64 table405)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child43)
     (served child64)
    )
  )
)
