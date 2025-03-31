; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 116905

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child470 child185 - child
    bread492 bread332 - bread-portion
    content460 content388 - content-portion
    tray281 tray383 tray255 tray477 - tray
    table49 table161 table493 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray281 kitchen)
     (at tray383 kitchen)
     (at tray255 kitchen)
     (at tray477 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_bread bread332)
     (at_kitchen_content content460)
     (at_kitchen_content content388)
     (no_gluten_bread bread332)
     (no_gluten_content content460)
     (allergic_gluten child470)
     (not_allergic_gluten child185)
     (waiting child470 table161)
     (waiting child185 table49)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child470)
     (served child185)
    )
  )
)
