; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 459471

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child443 child348 - child
    bread97 bread143 - bread-portion
    content264 content129 - content-portion
    tray369 - tray
    table44 table466 table335 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray369 kitchen)
     (at_kitchen_bread bread97)
     (at_kitchen_bread bread143)
     (at_kitchen_content content264)
     (at_kitchen_content content129)
     (no_gluten_bread bread97)
     (no_gluten_content content129)
     (allergic_gluten child443)
     (not_allergic_gluten child348)
     (waiting child443 table44)
     (waiting child348 table44)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child443)
     (served child348)
    )
  )
)
