; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 693354

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child307 child125 - child
    bread14 bread177 - bread-portion
    content447 content237 - content-portion
    tray321 tray29 - tray
    table1 table434 table417 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray321 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread14)
     (at_kitchen_bread bread177)
     (at_kitchen_content content447)
     (at_kitchen_content content237)
     (no_gluten_bread bread14)
     (no_gluten_content content237)
     (allergic_gluten child307)
     (not_allergic_gluten child125)
     (waiting child307 table434)
     (waiting child125 table434)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child307)
     (served child125)
    )
  )
)
