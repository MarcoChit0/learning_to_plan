; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 32538

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child162 child22 - child
    bread448 bread149 - bread-portion
    content119 content23 - content-portion
    tray253 tray51 tray427 - tray
    table449 table313 table100 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray253 kitchen)
     (at tray51 kitchen)
     (at tray427 kitchen)
     (at_kitchen_bread bread448)
     (at_kitchen_bread bread149)
     (at_kitchen_content content119)
     (at_kitchen_content content23)
     (no_gluten_bread bread149)
     (no_gluten_content content119)
     (allergic_gluten child22)
     (not_allergic_gluten child162)
     (waiting child162 table313)
     (waiting child22 table449)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child162)
     (served child22)
    )
  )
)
