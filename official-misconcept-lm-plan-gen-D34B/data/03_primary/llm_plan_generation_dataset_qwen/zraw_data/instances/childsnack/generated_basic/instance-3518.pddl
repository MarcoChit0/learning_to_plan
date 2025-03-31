; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 618846

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child295 child379 - child
    bread16 bread334 - bread-portion
    content412 content113 - content-portion
    tray95 tray446 - tray
    table50 table444 table473 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray95 kitchen)
     (at tray446 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_bread bread334)
     (at_kitchen_content content412)
     (at_kitchen_content content113)
     (not_allergic_gluten child379)
     (not_allergic_gluten child295)
     (waiting child295 table50)
     (waiting child379 table444)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child295)
     (served child379)
    )
  )
)
