; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 871894

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 child242 child459 - child
    bread170 bread493 bread320 - bread-portion
    content395 content427 content148 - content-portion
    tray108 tray390 tray71 - tray
    table46 table157 table161 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray108 kitchen)
     (at tray390 kitchen)
     (at tray71 kitchen)
     (at_kitchen_bread bread170)
     (at_kitchen_bread bread493)
     (at_kitchen_bread bread320)
     (at_kitchen_content content395)
     (at_kitchen_content content427)
     (at_kitchen_content content148)
     (not_allergic_gluten child459)
     (not_allergic_gluten child224)
     (not_allergic_gluten child242)
     (waiting child224 table161)
     (waiting child242 table157)
     (waiting child459 table161)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child224)
     (served child242)
     (served child459)
    )
  )
)
