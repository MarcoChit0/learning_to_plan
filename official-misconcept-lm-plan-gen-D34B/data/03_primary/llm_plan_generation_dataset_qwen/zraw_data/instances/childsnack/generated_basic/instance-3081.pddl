; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 805312

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 child195 - child
    bread395 bread289 - bread-portion
    content403 content433 - content-portion
    tray92 tray140 tray479 - tray
    table72 table22 table106 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray92 kitchen)
     (at tray140 kitchen)
     (at tray479 kitchen)
     (at_kitchen_bread bread395)
     (at_kitchen_bread bread289)
     (at_kitchen_content content403)
     (at_kitchen_content content433)
     (not_allergic_gluten child195)
     (not_allergic_gluten child224)
     (waiting child224 table106)
     (waiting child195 table106)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child224)
     (served child195)
    )
  )
)
