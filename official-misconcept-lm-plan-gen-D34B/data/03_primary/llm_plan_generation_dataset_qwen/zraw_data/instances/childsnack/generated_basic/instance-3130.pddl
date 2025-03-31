; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.3
; random seed: 515306

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 child144 - child
    bread359 bread322 - bread-portion
    content361 content40 - content-portion
    tray299 tray447 tray144 tray214 - tray
    table144 table41 table449 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray299 kitchen)
     (at tray447 kitchen)
     (at tray144 kitchen)
     (at tray214 kitchen)
     (at_kitchen_bread bread359)
     (at_kitchen_bread bread322)
     (at_kitchen_content content361)
     (at_kitchen_content content40)
     (no_gluten_bread bread322)
     (no_gluten_content content40)
     (allergic_gluten child92)
     (not_allergic_gluten child144)
     (waiting child92 table144)
     (waiting child144 table449)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child92)
     (served child144)
    )
  )
)
