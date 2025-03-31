; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.5
; random seed: 193423

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child3 child306 - child
    bread71 bread160 - bread-portion
    content224 content434 - content-portion
    tray490 tray323 tray148 tray197 - tray
    table85 table184 table234 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray490 kitchen)
     (at tray323 kitchen)
     (at tray148 kitchen)
     (at tray197 kitchen)
     (at_kitchen_bread bread71)
     (at_kitchen_bread bread160)
     (at_kitchen_content content224)
     (at_kitchen_content content434)
     (no_gluten_bread bread71)
     (no_gluten_content content224)
     (allergic_gluten child3)
     (not_allergic_gluten child306)
     (waiting child3 table234)
     (waiting child306 table234)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child3)
     (served child306)
    )
  )
)
