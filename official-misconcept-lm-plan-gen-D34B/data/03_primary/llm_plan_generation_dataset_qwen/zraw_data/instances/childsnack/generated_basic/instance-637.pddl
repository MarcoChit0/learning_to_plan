; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 502903

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 child338 - child
    bread119 bread142 - bread-portion
    content403 content155 - content-portion
    tray126 tray185 - tray
    table96 table205 table73 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at tray185 kitchen)
     (at_kitchen_bread bread119)
     (at_kitchen_bread bread142)
     (at_kitchen_content content403)
     (at_kitchen_content content155)
     (no_gluten_bread bread142)
     (no_gluten_content content403)
     (allergic_gluten child338)
     (not_allergic_gluten child170)
     (waiting child170 table73)
     (waiting child338 table96)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child170)
     (served child338)
    )
  )
)
