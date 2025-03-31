; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 548248

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child28 child453 - child
    bread183 bread141 - bread-portion
    content96 content7 - content-portion
    tray311 tray329 - tray
    table213 table130 table117 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray311 kitchen)
     (at tray329 kitchen)
     (at_kitchen_bread bread183)
     (at_kitchen_bread bread141)
     (at_kitchen_content content96)
     (at_kitchen_content content7)
     (no_gluten_bread bread141)
     (no_gluten_content content7)
     (allergic_gluten child453)
     (not_allergic_gluten child28)
     (waiting child28 table117)
     (waiting child453 table117)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child28)
     (served child453)
    )
  )
)
