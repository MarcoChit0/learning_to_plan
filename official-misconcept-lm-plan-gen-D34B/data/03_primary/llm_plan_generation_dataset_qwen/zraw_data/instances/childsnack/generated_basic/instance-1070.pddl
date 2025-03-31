; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 41177

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child374 child141 - child
    bread431 bread174 - bread-portion
    content150 content52 - content-portion
    tray36 tray232 tray29 - tray
    table168 table378 table415 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray36 kitchen)
     (at tray232 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread431)
     (at_kitchen_bread bread174)
     (at_kitchen_content content150)
     (at_kitchen_content content52)
     (not_allergic_gluten child141)
     (not_allergic_gluten child374)
     (waiting child374 table168)
     (waiting child141 table415)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child374)
     (served child141)
    )
  )
)
