; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 9166

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child57 child453 - child
    bread443 bread193 - bread-portion
    content380 content64 - content-portion
    tray436 tray32 - tray
    table459 table259 table445 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray436 kitchen)
     (at tray32 kitchen)
     (at_kitchen_bread bread443)
     (at_kitchen_bread bread193)
     (at_kitchen_content content380)
     (at_kitchen_content content64)
     (not_allergic_gluten child57)
     (not_allergic_gluten child453)
     (waiting child57 table259)
     (waiting child453 table459)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child57)
     (served child453)
    )
  )
)
