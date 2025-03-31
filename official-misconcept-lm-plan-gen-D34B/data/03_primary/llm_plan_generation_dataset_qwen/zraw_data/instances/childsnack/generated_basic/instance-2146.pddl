; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 831626

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child248 child428 - child
    bread301 bread456 - bread-portion
    content249 content299 - content-portion
    tray55 tray292 tray63 - tray
    table447 table460 table477 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray55 kitchen)
     (at tray292 kitchen)
     (at tray63 kitchen)
     (at_kitchen_bread bread301)
     (at_kitchen_bread bread456)
     (at_kitchen_content content249)
     (at_kitchen_content content299)
     (no_gluten_bread bread456)
     (no_gluten_content content249)
     (allergic_gluten child248)
     (not_allergic_gluten child428)
     (waiting child248 table460)
     (waiting child428 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child248)
     (served child428)
    )
  )
)
