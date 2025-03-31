; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 478946

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 child450 - child
    bread394 bread307 - bread-portion
    content304 content199 - content-portion
    tray159 tray118 tray76 tray49 - tray
    table418 table412 table334 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray159 kitchen)
     (at tray118 kitchen)
     (at tray76 kitchen)
     (at tray49 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread307)
     (at_kitchen_content content304)
     (at_kitchen_content content199)
     (no_gluten_bread bread307)
     (no_gluten_content content199)
     (allergic_gluten child450)
     (not_allergic_gluten child113)
     (waiting child113 table418)
     (waiting child450 table412)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child113)
     (served child450)
    )
  )
)
