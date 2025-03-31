; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 193961

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child25 child497 - child
    bread5 bread82 - bread-portion
    content102 content397 - content-portion
    tray241 tray282 - tray
    table410 table354 table164 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray241 kitchen)
     (at tray282 kitchen)
     (at_kitchen_bread bread5)
     (at_kitchen_bread bread82)
     (at_kitchen_content content102)
     (at_kitchen_content content397)
     (no_gluten_bread bread82)
     (no_gluten_content content102)
     (allergic_gluten child25)
     (not_allergic_gluten child497)
     (waiting child25 table164)
     (waiting child497 table410)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child25)
     (served child497)
    )
  )
)
