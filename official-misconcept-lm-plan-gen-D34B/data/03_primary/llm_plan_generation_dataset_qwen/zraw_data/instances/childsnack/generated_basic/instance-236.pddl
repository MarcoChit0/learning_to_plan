; child-snack task with 3 children and 0.74 gluten factor 
; constant factor of 1.3
; random seed: 440092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child322 child91 child223 - child
    bread231 bread427 bread429 - bread-portion
    content277 content285 content255 - content-portion
    tray327 - tray
    table482 table412 table235 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray327 kitchen)
     (at_kitchen_bread bread231)
     (at_kitchen_bread bread427)
     (at_kitchen_bread bread429)
     (at_kitchen_content content277)
     (at_kitchen_content content285)
     (at_kitchen_content content255)
     (no_gluten_bread bread231)
     (no_gluten_bread bread427)
     (no_gluten_content content277)
     (no_gluten_content content285)
     (allergic_gluten child223)
     (allergic_gluten child91)
     (not_allergic_gluten child322)
     (waiting child322 table482)
     (waiting child91 table482)
     (waiting child223 table482)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child322)
     (served child91)
     (served child223)
    )
  )
)
