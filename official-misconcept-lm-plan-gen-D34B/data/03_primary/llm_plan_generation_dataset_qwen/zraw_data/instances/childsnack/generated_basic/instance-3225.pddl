; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 686989

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child331 child32 - child
    bread238 bread38 - bread-portion
    content399 content330 - content-portion
    tray463 tray93 tray203 tray362 - tray
    table62 table41 table333 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray463 kitchen)
     (at tray93 kitchen)
     (at tray203 kitchen)
     (at tray362 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_bread bread38)
     (at_kitchen_content content399)
     (at_kitchen_content content330)
     (no_gluten_bread bread238)
     (no_gluten_content content330)
     (allergic_gluten child331)
     (not_allergic_gluten child32)
     (waiting child331 table333)
     (waiting child32 table41)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child331)
     (served child32)
    )
  )
)
