; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 779186

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child122 child294 child199 - child
    bread56 bread203 bread407 - bread-portion
    content45 content356 content146 - content-portion
    tray318 - tray
    table146 table203 table264 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray318 kitchen)
     (at_kitchen_bread bread56)
     (at_kitchen_bread bread203)
     (at_kitchen_bread bread407)
     (at_kitchen_content content45)
     (at_kitchen_content content356)
     (at_kitchen_content content146)
     (not_allergic_gluten child294)
     (not_allergic_gluten child122)
     (not_allergic_gluten child199)
     (waiting child122 table146)
     (waiting child294 table264)
     (waiting child199 table264)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child122)
     (served child294)
     (served child199)
    )
  )
)
