; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 727184

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child24 child371 - child
    bread406 bread264 - bread-portion
    content338 content90 - content-portion
    tray374 tray3 - tray
    table291 table488 table342 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray374 kitchen)
     (at tray3 kitchen)
     (at_kitchen_bread bread406)
     (at_kitchen_bread bread264)
     (at_kitchen_content content338)
     (at_kitchen_content content90)
     (not_allergic_gluten child371)
     (not_allergic_gluten child24)
     (waiting child24 table291)
     (waiting child371 table342)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child24)
     (served child371)
    )
  )
)
