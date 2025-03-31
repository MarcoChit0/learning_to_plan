; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 211575

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child435 child88 - child
    bread46 bread392 - bread-portion
    content207 content227 - content-portion
    tray318 tray427 - tray
    table2 table318 table219 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray318 kitchen)
     (at tray427 kitchen)
     (at_kitchen_bread bread46)
     (at_kitchen_bread bread392)
     (at_kitchen_content content207)
     (at_kitchen_content content227)
     (no_gluten_bread bread392)
     (no_gluten_content content207)
     (allergic_gluten child88)
     (not_allergic_gluten child435)
     (waiting child435 table2)
     (waiting child88 table219)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child435)
     (served child88)
    )
  )
)
