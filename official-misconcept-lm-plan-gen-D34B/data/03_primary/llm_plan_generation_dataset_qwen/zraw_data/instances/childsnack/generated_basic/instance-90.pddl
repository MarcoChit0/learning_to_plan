; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 445389

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 child10 - child
    bread485 bread181 - bread-portion
    content450 content231 - content-portion
    tray139 tray40 - tray
    table111 table18 table392 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray139 kitchen)
     (at tray40 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_bread bread181)
     (at_kitchen_content content450)
     (at_kitchen_content content231)
     (not_allergic_gluten child224)
     (not_allergic_gluten child10)
     (waiting child224 table392)
     (waiting child10 table18)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child224)
     (served child10)
    )
  )
)
