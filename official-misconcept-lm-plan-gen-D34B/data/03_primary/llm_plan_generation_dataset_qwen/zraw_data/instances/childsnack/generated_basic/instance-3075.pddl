; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 10639

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child138 child300 - child
    bread162 bread74 - bread-portion
    content178 content264 - content-portion
    tray323 tray189 tray183 - tray
    table26 table482 table419 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray323 kitchen)
     (at tray189 kitchen)
     (at tray183 kitchen)
     (at_kitchen_bread bread162)
     (at_kitchen_bread bread74)
     (at_kitchen_content content178)
     (at_kitchen_content content264)
     (no_gluten_bread bread162)
     (no_gluten_content content178)
     (allergic_gluten child300)
     (not_allergic_gluten child138)
     (waiting child138 table482)
     (waiting child300 table482)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child138)
     (served child300)
    )
  )
)
